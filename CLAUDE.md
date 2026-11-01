# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
bash setup.sh          # creates .venv, pip-installs requirements.txt, downloads DABench data
cp .env.example .env   # then fill in OPENAI_API_KEY and ANTHROPIC_API_KEY
```

All scripts must be run with the virtualenv interpreter:
```bash
.venv/bin/python scripts/<script>.py ...
```

## Commands

```bash
# Verify the full pipeline with no API keys (mock agent, no network calls)
.venv/bin/python scripts/smoke_test.py

# Run GPT-4o on the dev split (needs OPENAI_API_KEY)
.venv/bin/python scripts/run_experiment.py [--split dev] [--n 50] [--output data/results/run1]

# === JUDGING (rubric-based, 7-dimension; needs ANTHROPIC_API_KEY) ===

# PRODUCTION DEFAULT — blind-recompute judge: Opus solves the task BLIND, commits its
# answer, THEN sees the agent's work and judges. Best config by every metric.
.venv/bin/python scripts/run_blind_recompute_judge.py \
    --responses data/results/run1/agent_responses.jsonl \
    [--n N] [--output data/results/run1]

# Alternative judges:
#   rubric judge (reads code; optional independent-verifier signal):
.venv/bin/python scripts/run_rubric_judge.py \
    --responses data/results/run1/agent_responses.jsonl --mode no_reference \
    [--verifier data/results/run1/verifier_results.jsonl] [--prompt-version v1|v2]
#   independent verifier (Sonnet cold re-solve) → writes verifier_results.jsonl:
.venv/bin/python scripts/run_verifier.py --responses data/results/run1/agent_responses.jsonl
#   self-validating judge (Opus recomputes but sees agent code — anchors, weaker):
.venv/bin/python scripts/run_self_validating_judge.py --responses <...>

# Agreement vs the built-in answer key; head-to-head between two judge runs:
.venv/bin/python scripts/agreement_report.py --results <judge_results_*.jsonl> --label <name>
.venv/bin/python scripts/compare_self_validation.py \
    --new <judge_results_A.jsonl> --baseline <judge_results_B.jsonl> \
    --responses data/results/run1/agent_responses.jsonl

# Judge self-consistency (repeat-call stability: flips / score variance / confidence range):
.venv/bin/python scripts/run_consistency.py --responses <...> [--repeats 5]
.venv/bin/python scripts/consistency_report.py --results <consistency_results.jsonl>

# Legacy binary judge (0/1 scoring) + its calibration/analysis (pre-rubric generation):
.venv/bin/python scripts/run_judge.py --responses <...> [--judge programmatic|composite|llm]
.venv/bin/python scripts/calibrate.py --responses <...> [--n 100] [--seed 42]
.venv/bin/python scripts/analyze_results.py --scores data/results/run1/judge_scores.jsonl
```

## Architecture

### Data format

`data/dabench/infiagent-repo/examples/DA-Agent/data/` contains:
- `da-dev-questions.jsonl` — 257 tasks, fields: `id`, `question`, `concepts`, `constraints`, `format`, `file_name`, `level`
- `da-dev-labels.jsonl` — ground truth, fields: `id`, `common_answers` (list of `[answer_name, answer_value]` pairs)
- `da-dev-tables/<file_name>` — CSV files referenced by each task

`DABenchLoader(data_dir=Path("data/dabench")).load_tasks("dev")` merges both JSONL files and resolves each task's CSV to an absolute `file_path`. The resulting `DABenchTask` objects carry `common_answers` from the labels file (or `None` for withheld test splits).

### Answer encoding: `@name[value]`

The benchmark encodes structured answers in stdout using the pattern `@answer_name[answer_value]`. The agent's generated code must `print(f"@mean_fare[{result}]")` — anything else in stdout is ignored. The regex `r"@(\w+)\[(.*?)\]"` (in `openai_agent.py`) parses all such tokens into `AgentResponse.answer` (a `dict[str, str]`). The `format` field of each task specifies exactly which answer names are expected.

### Pipeline: benchmark → agent → judge

```
src/benchmark/loader.py    DABenchLoader.load_tasks()  →  list[DABenchTask]
src/agents/openai_agent.py OpenAIAgent.solve(task)     →  AgentResponse  (the agent UNDER TEST = GPT-4o)
src/agents/executor.py     CodeExecutor.run(code, dir) →  (stdout, stderr)
src/benchmark/evaluator.py infiagent_score(task,resp) →  float 0/1  (the built-in answer-key check)
src/experiments/runner.py  ExperimentRunner.run(tasks) →  list[AgentResponse]
```

`ExperimentRunner` appends results to `agent_responses.jsonl` immediately after each task. On restart it reads completed task IDs from the file and skips them, making runs fully resumable. (The rubric-judge scripts in `scripts/` follow the same resume-by-task-id pattern.)

### Judge variants (production default: blind-recompute)

Two generations of judges live in `src/judge/`:
- **Binary (legacy):** `ProgrammaticJudge` / `ClaudeJudge` / `CompositeJudge` → `JudgeScore` (0/1). Driven by `run_judge.py` + `calibrate.py`.
- **Rubric (current):** Opus 4.7 scores 7 dimensions 1-5 + `critical_errors` + pass/fail via tool-use. A deterministic layer (`src/judge/scoring.py`: `compute_process_quality_score`, `compute_overall_pass`, `classify_run`) recomputes the verdict from the scores — the LLM's own `overall_pass` is stored but not trusted. `src/judge/schema.py` (CSV summary) and `src/judge/prechecks.py` feed every rubric judge.

Rubric-judge configs, by no-reference Cohen's κ vs the built-in key on the 40-run batch:

| config | entry point | κ | recall |
|---|---|---|---|
| plain rubric | `run_rubric_judge.py` | 0.23 | 0.50 |
| + independent verifier | `run_rubric_judge.py --verifier …` | 0.27 | 0.67 |
| self-validating (code-aware) | `run_self_validating_judge.py` | 0.27 | 0.50 |
| **blind-recompute (DEFAULT)** | **`run_blind_recompute_judge.py`** | **0.59** | **0.83** |

**`src/judge/blind_recompute_judge.py` is the default.** Two phases: (1) Opus solves the task BLIND — sees only question/constraints/format/schema + a `run_python` tool, *never* the agent's code/answer/prechecks — and commits its answer via `submit_blind_answer`; (2) the agent's artifacts are revealed and it judges by comparison. Committing blind preserves independence, which both catches subtle wrong answers (recall↑) *and* vindicates correct-but-weird-process answers (precision↑). It beats even the separate-model verifier; costs ~3-4 Opus calls/task.

**Independence is the active ingredient** — the verifier (`src/judge/verifier.py`, a Sonnet cold re-solve) and blind-recompute work for the same reason; the code-aware self-validating judge underperforms precisely because seeing the agent's code makes its recompute anchor to the agent.

**Opus 4.7 rejects `temperature`/`top_p`/`top_k`** — all judges keep the param for interface compatibility but never send it to the API.

### Ground truth location

`common_answers` is a list of `[name, value]` pairs loaded from `da-dev-labels.jsonl` at load time. `ProgrammaticJudge` iterates these pairs, looks up each name in `AgentResponse.answer` (case-insensitive), and applies 1% relative numeric tolerance (or ±0.5 for near-zero expected values) before falling back to string comparison. All pairs must match for a score of 1.0.

### Calibration

`scripts/calibrate.py` treats `ProgrammaticJudge` as the oracle (ground truth) and measures how well `ClaudeJudge` agrees with it. `compute_calibration(judge_scores, ground_truth)` returns accuracy, precision, recall, F1, and Cohen's kappa. `bootstrap_ci(scores)` adds a percentile bootstrap confidence interval around Claude judge accuracy.

### Config

`config.yaml` holds defaults for all scripts. Scripts use `argparse` to override individual flags; PyYAML loads the config and `python-dotenv` loads `.env` for API keys.
