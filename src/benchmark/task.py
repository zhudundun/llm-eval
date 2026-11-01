"""
Dataclasses representing DABench tasks, agent responses, and judge scores.

Field names are derived directly from the InfiAgent-DABench JSONL schema:
  questions: id, question, concepts, constraints, format, file_name, level
  labels:    id, common_answers  (list of [answer_name, answer_value] pairs)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DABenchTask:
    """
    One question from da-dev-questions.jsonl merged with its label from
    da-dev-labels.jsonl.

    Field names match the JSONL keys exactly.  ``file_path`` is the resolved
    absolute path to the associated CSV file (None when the file is absent).
    """

    # --- question fields (from da-dev-questions.jsonl) ---
    id: int
    question: str
    # Ordered list of skill categories, e.g. ["Summary Statistics", "Feature Engineering"]
    concepts: list[str]
    # Free-text constraints on how to solve the task
    constraints: str
    # Answer-format specification string, e.g. "@mean_fare[mean_fare_value] where ..."
    format: str
    # Bare filename of the associated CSV, e.g. "test_ave.csv"
    file_name: str
    # Difficulty tier: "easy" | "medium" | "hard"
    level: str

    # --- label fields (from da-dev-labels.jsonl) ---
    # List of [answer_name, answer_value] pairs, e.g. [["mean_fare", "34.65"]]
    # None when the split has no ground-truth labels (e.g. a held-out test set).
    common_answers: Optional[list[list[str]]] = None

    # --- resolved by the loader ---
    # Absolute path to the CSV on disk; None if the file is missing.
    file_path: Optional[Path] = None


@dataclass
class AgentResponse:
    """
    The output produced by a model/agent for a single DABench task.

    ``answer`` is the parsed structured answer extracted from ``stdout``
    (i.e. the @tag[value] pairs), represented as a dict mapping answer-name
    to its string value.
    """

    # Identity
    task_id: int
    dataset_id: str          # = task.file_name (the CSV filename)
    question: str            # = task.question (denormalized for self-contained rows)

    # Agent
    model: str
    reasoning_trace: Optional[str]   # full raw LLM response text (pre-code, chain-of-thought)
    code: str                        # extracted Python code block
    attempts: int

    # Execution
    stdout: str
    stderr: str
    execution_result: str    # "success" | "error" | "timeout" | "no_answer"

    # Answers
    answer: dict             # parsed {name: value} from @name[value] in stdout

    # Evaluation
    builtin_eval_result: Optional[float]   # InfiAgent's own scorer (0.0 or 1.0); None if ground truth absent

    # Cost / perf
    runtime_seconds: float
    token_usage: dict        # {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N}

    # Error
    error: Optional[str]     # first stderr line or exception message; None on success


@dataclass
class JudgeScore:
    """
    The score assigned to one AgentResponse by a judge.

    ``score`` is 0.0 (wrong) or 1.0 (correct).
    ``method`` is "programmatic" when the answer was compared directly with
    the ground-truth label, or "llm" when an LLM was used to adjudicate.
    """

    task_id: int
    judge_model: str
    score: float          # 0.0 or 1.0
    reasoning: str
    method: str           # "programmatic" | "llm"


@dataclass
class CriticalError:
    type: str
    severity: str          # "low" | "medium" | "high"
    explanation: str


@dataclass
class RubricJudgeResult:
    """One LLM-judge verdict for an AgentResponse, plus deterministic-layer fields filled in later."""
    task_id: int
    agent_model: str
    judge_model: str
    judge_prompt_version: str
    judge_mode: str                 # "no_reference" | "with_reference"
    # raw judge_result fields:
    task_success_score: int
    code_correctness_score: int
    data_handling_score: int
    statistical_validity_score: int
    reproducibility_score: int
    final_answer_quality_score: int
    reasoning_quality_score: int
    critical_errors: list           # list[dict] of {type, severity, explanation}
    overall_pass: bool
    confidence: float
    judge_explanation: str
    parse_success: bool
    # deterministic layer (filled by scoring.py downstream; default None/0):
    computed_process_quality_score: Optional[float] = None
    computed_overall_pass: Optional[bool] = None
    builtin_pass: Optional[bool] = None
    classification: Optional[str] = None


@dataclass
class VerifierResult:
    """Output of the independent reference-free verifier for one AgentResponse.

    The verifier re-derives the answer COLD with a different-family model
    (it never sees the agent's code/answer or any reference), then compares the
    two and separately re-runs the agent's own code to check reproducibility.
    """

    task_id: int
    verifier_model: str

    # --- independent re-solve ---
    verifier_code: str
    verifier_stdout: str
    verifier_stderr: str
    verifier_answer: dict            # {name: value} parsed from verifier stdout
    verifier_execution_ok: bool      # verifier code ran AND produced a non-empty answer

    # --- comparison: verifier vs agent ---
    agrees_with_agent: Optional[bool]   # True=all shared answers match; False=any mismatch; None=can't compare
    agreement_detail: str               # human-readable per-field: "name: agent=X verifier=Y rel=..%"

    # --- re-execution of the AGENT's own code (reproducibility) ---
    reproduced_answer: dict             # {name: value} from re-running agent code
    agent_code_reproduces: Optional[bool]  # re-run stdout reproduces agent's claimed answer; None if agent code missing

    # --- cost ---
    runtime_seconds: float
    token_usage: dict
    error: Optional[str]
