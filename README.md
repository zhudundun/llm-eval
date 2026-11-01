# LLM Eval — LLM-as-Judge Evaluation Framework

An evaluation framework for benchmarking data-analysis LLM agents on [DABench](https://github.com/InfiAgent/InfiAgent) (257 tasks). Runs GPT-4o as the agent under test and Claude Opus as the judge, with four judge variants studied for agreement vs the built-in answer key.

**Key finding: blind-recompute judge achieves Cohen's κ = 0.59 and 0.83 recall** — beating plain rubric (κ = 0.23), rubric + verifier (κ = 0.27), and self-validating judge (κ = 0.27). Independence is the active ingredient.

---

## Architecture

```
DABench tasks (257)
      │
      ▼
┌─────────────────┐
│  OpenAI Agent   │  GPT-4o generates Python code → executes → parses @answer[value] tokens
│  (under test)   │
└─────────────────┘
      │  agent_responses.jsonl
      ▼
┌─────────────────┐
│     Judge       │  Claude Opus scores 7 rubric dimensions (1–5) + pass/fail
│   (4 variants)  │  Deterministic scoring layer recomputes verdict from scores
└─────────────────┘
      │  judge_results.jsonl
      ▼
┌─────────────────┐
│    Analysis     │  Agreement vs answer key, head-to-head, consistency, calibration
└─────────────────┘
```

### Answer encoding

Agents must emit structured answers as `@answer_name[answer_value]` tokens in stdout — e.g. `print(f"@mean_fare[{result:.4f}]")`. Anything else in stdout is ignored. The `format` field of each task specifies exactly which answer names are expected.

---

## Judge variants

| Judge | κ vs answer key | Recall | Entry point |
|---|---|---|---|
| Plain rubric | 0.23 | 0.50 | `run_rubric_judge.py` |
| Rubric + independent verifier | 0.27 | 0.67 | `run_rubric_judge.py --verifier` |
| Self-validating (code-aware) | 0.27 | 0.50 | `run_self_validating_judge.py` |
| **Blind-recompute (default)** | **0.59** | **0.83** | `run_blind_recompute_judge.py` |

**Blind-recompute:** Opus solves the task completely blind (sees only question/constraints/format/schema + a `run_python` tool, never the agent's code/answer). It commits its own answer via `submit_blind_answer`, then the agent's artifacts are revealed for phase-2 judging. Committing blind preserves independence — this both catches subtle wrong answers (recall ↑) and vindicates correct-but-weird-process answers (precision ↑).

The self-validating judge underperforms precisely because seeing the agent's code anchors its recomputation to the agent.

---

## Setup

```bash
bash setup.sh                  # creates .venv, installs requirements.txt, downloads DABench
cp .env.example .env           # then fill in API keys
```

```bash
# Verify the full pipeline with no API keys
.venv/bin/python scripts/smoke_test.py
```

Requires Python 3.11+. The DABench download (~111 MB) happens inside `setup.sh`.

---

## Running experiments

All scripts use the virtualenv interpreter and accept `--help` for full options.

```bash
# 1. Run GPT-4o on the dev split (needs OPENAI_API_KEY)
.venv/bin/python scripts/run_experiment.py --split dev --n 50 --output data/results/run1

# 2. Judge with blind-recompute (PRODUCTION DEFAULT; needs ANTHROPIC_API_KEY)
.venv/bin/python scripts/run_blind_recompute_judge.py \
    --responses data/results/run1/agent_responses.jsonl \
    --output data/results/run1

# 3. Agreement vs built-in answer key
.venv/bin/python scripts/agreement_report.py \
    --results data/results/run1/judge_results_blind_recompute.jsonl \
    --label blind_recompute
```

Runs are **fully resumable** — the runner reads completed task IDs from the output JSONL and skips them on restart.

### Alternative judges

```bash
# Rubric judge (reads agent code; add --verifier for independent-verifier signal)
.venv/bin/python scripts/run_rubric_judge.py \
    --responses data/results/run1/agent_responses.jsonl --mode no_reference

# Independent verifier (Sonnet cold re-solve) → writes verifier_results.jsonl
.venv/bin/python scripts/run_verifier.py \
    --responses data/results/run1/agent_responses.jsonl

# Self-validating judge (Opus recomputes but sees agent code — weaker)
.venv/bin/python scripts/run_self_validating_judge.py \
    --responses data/results/run1/agent_responses.jsonl
```

### Consistency and calibration

```bash
# Judge self-consistency (repeat-call score variance)
.venv/bin/python scripts/run_consistency.py --responses data/results/run1/agent_responses.jsonl
.venv/bin/python scripts/consistency_report.py --results data/results/run1/consistency_results.jsonl

# Head-to-head between two judge runs
.venv/bin/python scripts/compare_self_validation.py \
    --new data/results/run1/judge_results_A.jsonl \
    --baseline data/results/run1/judge_results_B.jsonl \
    --responses data/results/run1/agent_responses.jsonl
```

---

## Project layout

```
src/
  agents/
    openai_agent.py      # GPT-4o agent — generates + executes Python, parses @answer tokens
    executor.py          # sandboxed Python code executor
  benchmark/
    loader.py            # DABenchLoader — merges questions + labels, resolves CSV paths
    task.py              # DABenchTask dataclass
    evaluator.py         # infiagent_score() — built-in answer-key check
  judge/
    blind_recompute_judge.py   # DEFAULT — 2-phase: blind solve → reveal + judge
    rubric_judge.py            # 7-dimension rubric judge (read agent code)
    self_validating_judge.py   # Opus recomputes with agent code visible
    verifier.py                # Sonnet cold re-solve (independent verifier)
    scoring.py                 # deterministic: compute_process_quality_score, classify_run
    schema.py                  # CSV prechecks summary builder
    prechecks.py               # structural checks run before judging
    programmatic.py            # ProgrammaticJudge (numeric + string tolerance)
    composite.py               # CompositeJudge (programmatic + LLM)
    claude_judge.py            # legacy binary ClaudeJudge
    base.py                    # JudgeScore / RubricResult dataclasses
    prompts/                   # versioned judge prompt templates (.txt)
  experiments/
    runner.py            # ExperimentRunner — resumable, appends-per-task
  reliability/
    consistency.py       # repeat-call stability measurement
    bootstrap.py         # percentile bootstrap CI
  calibration/
    agreement.py         # Cohen's κ, precision/recall/F1 vs answer key
    metrics.py           # compute_calibration, bootstrap_ci
scripts/
  run_experiment.py            # drive GPT-4o on DABench
  run_blind_recompute_judge.py # blind-recompute judge runner
  run_rubric_judge.py          # rubric judge runner
  run_self_validating_judge.py # self-validating judge runner
  run_verifier.py              # independent verifier runner
  run_consistency.py           # consistency measurement
  agreement_report.py          # agreement vs answer key
  consistency_report.py        # consistency report
  compare_self_validation.py   # head-to-head judge comparison
  calibrate.py                 # calibration vs programmatic oracle
  analyze_results.py           # result analysis
  smoke_test.py                # no-API end-to-end smoke test
  test_e2e.py                  # full end-to-end test
data/
  dabench/             # downloaded by setup.sh (gitignored — run setup.sh to regenerate)
  results/             # experiment run outputs — gitignored
reports/               # generated analysis reports — gitignored
```

---

## Configuration

`config.yaml` holds defaults for all scripts. Override individual flags via argparse. API keys go in `.env` (never committed — use `.env.example` as template).

| Key | Default | Description |
|---|---|---|
| `agent.model` | `gpt-4o` | Agent model under test |
| `judge.default` | `blind_recompute` | Active judge variant |
| `judge.model` | `claude-opus-4-7` | Judge model |
| `experiment.split` | `dev` | DABench split (`dev` = 257 tasks) |
| `reliability.n_bootstrap` | `1000` | Bootstrap CI samples |

---

## Environment variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Required for `run_experiment.py` |
| `ANTHROPIC_API_KEY` | Required for all judge scripts |

Copy `.env.example` to `.env` and fill in — never commit `.env`.
