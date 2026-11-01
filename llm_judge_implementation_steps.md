# Implementation Steps for Designing an LLM Judge for Agentic Data Science

## 1. Goal

Design an LLM-as-a-judge system that evaluates data-science agent outputs for:

- Final-answer correctness
- Code correctness
- Data handling quality
- Statistical validity
- Reproducibility
- Reasoning quality
- Unsupported or hallucinated claims
- Failure modes such as wrong filters, wrong aggregation, bad joins, or code execution errors

For benchmark development, use InfiAgent-DABench as a controlled lab because it can provide reference answers and built-in evaluation results. For real-world use, run the judge without reference answers and focus on support, consistency, reproducibility, and process quality.

---

## 2. Core Design Principle

The judge should not only answer:

> Did the final answer match the reference?

That is what a benchmark evaluator already does.

The judge should answer:

> Was the agent's solution process correct, reproducible, and supported by the data?

Therefore, the judge must evaluate artifacts, not just the final answer.

Important artifacts include:

- User question
- Dataset schema or summary
- Sample rows
- Agent plan, if available
- Agent-generated code
- Execution stdout
- Execution stderr
- Final answer
- Deterministic pre-checks
- Reference answer, if available in benchmark mode
- Built-in evaluator result, if available in benchmark mode

---

## 3. Judge Modes

Implement three judge modes.

### 3.1 No-reference mode

This is the real-world production mode.

Inputs:

- Question
- Dataset summary
- Agent code
- Execution logs
- Final answer
- Pre-check results

No reference answer is provided.

The judge evaluates whether the answer is supported by the code, logs, and dataset structure.

### 3.2 Reference-aware mode

This is the benchmark calibration mode.

Inputs:

- Everything from no-reference mode
- Reference answer
- Built-in evaluator result

The judge evaluates both final-answer correctness and process quality.

### 3.3 Pairwise mode

This is the regression-testing mode.

Inputs:

- Same task
- Output from Agent A
- Output from Agent B
- Code/logs/final answer for each agent

The judge decides which output is better supported and more reliable.

---

## 4. System Architecture

```text
InfiAgent-DABench / real data task
        |
        v
Agent under test
        |
        v
Run artifacts
  - final answer
  - code
  - stdout
  - stderr
  - runtime metadata
        |
        v
Deterministic pre-checks
        |
        v
LLM judge
        |
        v
Structured judge result
        |
        v
Agreement analysis / regression analysis / human-review triage
```

---

## 5. Implementation Folder Structure

```text
infiagent_judge/
  README.md

  data/
    agent_runs.jsonl
    judge_results.jsonl
    human_labels.csv

  prompts/
    judge_no_reference_v1.txt
    judge_with_reference_v1.txt
    pairwise_judge_v1.txt

  src/
    extract_schema.py
    prechecks.py
    prompt_builder.py
    llm_client.py
    judge_runner.py
    validate_output.py
    scoring.py
    analyze_agreement.py

  reports/
    agreement_report.md
    suspicious_passes.md
    near_misses.md
```

---

## 6. Step 1: Collect Agent Run Artifacts

Each agent run should be saved as one JSONL row.

Example schema:

```json
{
  "task_id": "task_001",
  "question": "Which city has the highest average sales?",
  "csv_path": "data/sales.csv",
  "agent_model": "gpt-4.1-mini",
  "agent_prompt_version": "react_v1",
  "agent_final_answer": "San Jose",
  "agent_code": "import pandas as pd\n...",
  "stdout": "...",
  "stderr": "...",
  "execution_success": true,
  "runtime_sec": 12.3,
  "input_tokens": 1234,
  "output_tokens": 567,
  "cost_usd": 0.01,
  "reference_answer": "San Jose",
  "builtin_pass": true
}
```

For real-world production use, `reference_answer` and `builtin_pass` will usually be null or absent.

---

## 7. Step 2: Extract Dataset Summary

Create `extract_schema.py` to summarize CSV files.

The judge usually does not need the full dataset. It needs a compact summary.

Example summary:

```json
{
  "num_rows": 1200,
  "num_columns": 6,
  "columns": [
    {
      "name": "city",
      "dtype": "object",
      "num_missing": 0,
      "num_unique": 45,
      "sample_values": ["San Jose", "Seattle", "Austin"]
    },
    {
      "name": "sales",
      "dtype": "float64",
      "num_missing": 12,
      "min": 5.2,
      "max": 999.4,
      "mean": 142.8
    }
  ],
  "sample_rows": [
    {"city": "San Jose", "sales": 120.5},
    {"city": "Seattle", "sales": 98.2}
  ]
}
```

This helps the judge detect:

- Nonexistent columns
- Wrong dtypes
- Missing-value issues
- Suspicious assumptions
- Claims unsupported by the available data

---

## 8. Step 3: Implement Deterministic Pre-checks

Create `prechecks.py`.

These checks are cheap, deterministic, and help ground the LLM judge.

Suggested checks:

- Did the code execute successfully?
- Is stderr non-empty?
- Is the final answer empty?
- Is code missing?
- Did the code reference unknown columns?
- Did the code use the expected CSV file?
- Did the final answer appear to be hardcoded?
- Did the answer include a number/string not present in stdout?
- Did the code timeout?

Example pre-check output:

```json
{
  "execution_success": false,
  "stderr_present": true,
  "final_answer_empty": false,
  "code_present": true,
  "referenced_unknown_columns": ["revenu"],
  "possible_hardcoded_answer": false
}
```

Example function skeleton:

```python
def run_prechecks(run, dataset_summary):
    return {
        "execution_success": run.get("execution_success", False),
        "stderr_present": bool(run.get("stderr", "").strip()),
        "final_answer_empty": not bool(run.get("agent_final_answer", "").strip()),
        "code_present": bool(run.get("agent_code", "").strip()),
        "referenced_unknown_columns": detect_unknown_columns(
            run.get("agent_code", ""),
            dataset_summary
        ),
        "possible_hardcoded_answer": detect_hardcoded_answer(
            run.get("agent_code", ""),
            run.get("agent_final_answer", "")
        )
    }
```

---

## 9. Step 4: Define the Judge Rubric

Use a 1-5 score for each dimension.

```text
1 = wrong or unusable
2 = major issues
3 = partially correct
4 = mostly correct
5 = correct and well-supported
```

Recommended scoring dimensions:

| Field | Meaning |
|---|---|
| `task_success_score` | Did the final answer solve the task? |
| `code_correctness_score` | Was the code logically correct and executable? |
| `data_handling_score` | Did the agent load, filter, aggregate, and transform data correctly? |
| `statistical_validity_score` | Did it use appropriate math/statistics? |
| `reproducibility_score` | Could the answer be reproduced from the code and logs? |
| `final_answer_quality_score` | Was the answer clear, formatted correctly, and not over-explained? |
| `reasoning_quality_score` | Was the method reasonable and supported? |

---

## 10. Step 5: Define Failure Taxonomy

Use a controlled list of failure types so results can be aggregated.

```python
FAILURE_TYPES = [
    "wrong_answer",
    "wrong_filter",
    "wrong_aggregation",
    "wrong_join",
    "wrong_groupby",
    "wrong_sorting",
    "wrong_column",
    "wrong_unit",
    "rounding_error",
    "missing_value_error",
    "data_leakage",
    "code_execution_error",
    "timeout",
    "hallucinated_column",
    "unsupported_claim",
    "incomplete_answer",
    "format_error"
]
```

---

## 11. Step 6: Define Judge Output Schema

The judge must return strict JSON.

```json
{
  "task_success_score": 5,
  "code_correctness_score": 5,
  "data_handling_score": 4,
  "statistical_validity_score": 4,
  "reproducibility_score": 5,
  "final_answer_quality_score": 5,
  "reasoning_quality_score": 4,
  "critical_errors": [],
  "overall_pass": true,
  "confidence": 0.91,
  "judge_explanation": "The agent correctly grouped by city, computed the average sales, and returned the city with the highest value."
}
```

Failure example:

```json
{
  "task_success_score": 1,
  "code_correctness_score": 2,
  "data_handling_score": 1,
  "statistical_validity_score": 3,
  "reproducibility_score": 2,
  "final_answer_quality_score": 2,
  "reasoning_quality_score": 2,
  "critical_errors": [
    {
      "type": "wrong_column",
      "severity": "high",
      "explanation": "The task asked for total revenue, but the code grouped by profit."
    }
  ],
  "overall_pass": false,
  "confidence": 0.87,
  "judge_explanation": "The final answer is not reliable because the analysis used the wrong metric."
}
```

---

## 12. Step 7: Write Judge Prompt Templates

### 12.1 No-reference judge prompt

Use this in real-world mode.

```text
You are an expert data-science evaluator.

Evaluate the AI agent's solution to a data-analysis task.

You must judge whether the agent's final answer is supported by its code, execution logs, and the dataset schema.

Be strict. Do not give credit for plausible claims that are not supported by code or output.

Task:
{question}

Dataset summary:
{dataset_summary}

Pre-check results:
{precheck_results}

Agent final answer:
{agent_final_answer}

Agent code:
{agent_code}

Execution stdout:
{stdout}

Execution stderr:
{stderr}

Return only valid JSON with this schema:
{
  "task_success_score": integer 1-5,
  "code_correctness_score": integer 1-5,
  "data_handling_score": integer 1-5,
  "statistical_validity_score": integer 1-5,
  "reproducibility_score": integer 1-5,
  "final_answer_quality_score": integer 1-5,
  "reasoning_quality_score": integer 1-5,
  "critical_errors": [
    {
      "type": one of [
        "wrong_answer",
        "wrong_filter",
        "wrong_aggregation",
        "wrong_join",
        "wrong_groupby",
        "wrong_sorting",
        "wrong_column",
        "wrong_unit",
        "rounding_error",
        "missing_value_error",
        "data_leakage",
        "code_execution_error",
        "timeout",
        "hallucinated_column",
        "unsupported_claim",
        "incomplete_answer",
        "format_error"
      ],
      "severity": "low" | "medium" | "high",
      "explanation": string
    }
  ],
  "overall_pass": boolean,
  "confidence": number between 0 and 1,
  "judge_explanation": string
}
```

### 12.2 Reference-aware judge prompt

Use this in benchmark mode.

Add this section to the no-reference prompt:

```text
Reference answer:
{reference_answer}

Built-in evaluator result:
{builtin_result}

Use the reference answer to assess final correctness, but still evaluate whether the process was valid.
```

### 12.3 Pairwise judge prompt

Use this for regression testing.

```text
You are an expert data-science evaluator.

Two agents attempted the same data-analysis task. Decide which solution is better supported by the code, execution logs, and final answer.

Task:
{question}

Dataset summary:
{dataset_summary}

Agent A final answer:
{agent_a_final_answer}

Agent A code:
{agent_a_code}

Agent A stdout/stderr:
{agent_a_logs}

Agent B final answer:
{agent_b_final_answer}

Agent B code:
{agent_b_code}

Agent B stdout/stderr:
{agent_b_logs}

Return only valid JSON:
{
  "winner": "A" | "B" | "tie",
  "confidence": number between 0 and 1,
  "a_strengths": [string],
  "a_weaknesses": [string],
  "b_strengths": [string],
  "b_weaknesses": [string],
  "reason": string
}
```

---

## 13. Step 8: Implement Judge Model Caller

Create `llm_client.py` and `judge_runner.py`.

Recommended settings:

```python
temperature = 0
max_tokens = 1000
```

Use structured output or JSON mode if supported by the model provider.

Example interface:

```python
class LLMJudge:
    def __init__(self, model_name, client, prompt_template):
        self.model_name = model_name
        self.client = client
        self.prompt_template = prompt_template

    def judge(self, judge_input):
        prompt = self.prompt_template.format(**judge_input)
        raw = self.call_model(prompt)
        parsed = self.parse_json(raw)
        validated = self.validate(parsed)
        return validated

    def call_model(self, prompt):
        raise NotImplementedError
```

---

## 14. Step 9: Implement JSON Parser and Validator

Never trust raw LLM output.

Create `validate_output.py`.

```python
def validate_judge_output(obj):
    required = [
        "task_success_score",
        "code_correctness_score",
        "data_handling_score",
        "statistical_validity_score",
        "reproducibility_score",
        "final_answer_quality_score",
        "reasoning_quality_score",
        "critical_errors",
        "overall_pass",
        "confidence",
        "judge_explanation"
    ]

    for key in required:
        if key not in obj:
            raise ValueError(f"Missing required field: {key}")

    score_fields = [
        "task_success_score",
        "code_correctness_score",
        "data_handling_score",
        "statistical_validity_score",
        "reproducibility_score",
        "final_answer_quality_score",
        "reasoning_quality_score"
    ]

    for field in score_fields:
        if not isinstance(obj[field], int):
            raise ValueError(f"{field} must be int")
        if obj[field] < 1 or obj[field] > 5:
            raise ValueError(f"{field} must be between 1 and 5")

    if not isinstance(obj["overall_pass"], bool):
        raise ValueError("overall_pass must be boolean")

    if not 0 <= obj["confidence"] <= 1:
        raise ValueError("confidence must be between 0 and 1")

    return obj
```

Add retry logic:

```text
1st call: normal judge prompt
2nd call: JSON repair prompt
3rd failure: mark parse_success = false
```

---

## 15. Step 10: Implement Deterministic Scoring Layer

Do not fully trust the LLM's `overall_pass`. Store it, but also compute your own pass/fail rule.

Create `scoring.py`.

```python
def compute_process_quality_score(r):
    return (
        0.30 * r["task_success_score"] +
        0.25 * r["code_correctness_score"] +
        0.20 * r["data_handling_score"] +
        0.10 * r["statistical_validity_score"] +
        0.10 * r["reproducibility_score"] +
        0.05 * r["final_answer_quality_score"]
    )


def compute_overall_pass(r):
    high_severity_errors = [
        e for e in r["critical_errors"]
        if e.get("severity") == "high"
    ]

    if high_severity_errors:
        return False

    required_scores = [
        r["task_success_score"],
        r["code_correctness_score"],
        r["data_handling_score"],
        r["final_answer_quality_score"]
    ]

    return min(required_scores) >= 4


def classify_run(builtin_pass, judge_pass, process_score):
    if builtin_pass is True and judge_pass and process_score >= 4.0:
        return "strong_pass"
    if builtin_pass is True and not judge_pass:
        return "suspicious_pass"
    if builtin_pass is False and process_score >= 3.5:
        return "near_miss"
    if builtin_pass is False and process_score < 3.5:
        return "clear_fail"
    if builtin_pass is None and judge_pass:
        return "judge_pass_no_reference"
    return "judge_fail_no_reference"
```

---

## 16. Step 11: Store Judge Results

Save each judge result as JSONL.

Example:

```json
{
  "task_id": "task_001",
  "agent_model": "gpt-4.1-mini",
  "judge_model": "gpt-4.1",
  "judge_prompt_version": "judge_v1_no_reference",
  "judge_mode": "no_reference",
  "judge_result": {
    "task_success_score": 5,
    "code_correctness_score": 5,
    "data_handling_score": 4,
    "statistical_validity_score": 4,
    "reproducibility_score": 5,
    "final_answer_quality_score": 5,
    "reasoning_quality_score": 4,
    "critical_errors": [],
    "overall_pass": true,
    "confidence": 0.91,
    "judge_explanation": "..."
  },
  "computed_process_quality_score": 4.65,
  "computed_overall_pass": true,
  "builtin_pass": true,
  "classification": "strong_pass",
  "parse_success": true
}
```

---

## 17. Step 12: Implement Agreement Analyzer

Create `analyze_agreement.py`.

Compare:

- Built-in evaluator vs LLM judge
- LLM judge vs human labels
- Built-in evaluator vs human labels
- Old agent vs new agent using pairwise judge

Minimum metrics:

- Accuracy
- Precision
- Recall
- F1
- Confusion matrix
- Cohen's kappa
- Spearman correlation for 1-5 scores

Useful report sections:

```text
Total tasks: 100
Built-in pass rate: 62%
Judge pass rate: 58%
Agreement: 84%

Confusion matrix:
                 Built-in Pass   Built-in Fail
Judge Pass       54              4
Judge Fail       8               34

Suspicious passes: 8
Near misses: 4

Top failure types:
wrong_aggregation: 12
wrong_filter: 8
rounding_error: 6
format_error: 5
```

---

## 18. Step 13: Build Human Calibration Set

Create a small labeled set of 20-50 examples.

Example `human_labels.csv`:

```csv
task_id,human_pass,human_task_success,human_code_correctness,human_data_handling,human_notes
task_001,true,5,5,4,"Correct grouping and final answer"
task_002,false,2,3,1,"Filtered wrong year"
```

Use this to answer:

- Does the judge agree with humans?
- Does the built-in evaluator agree with humans?
- Are judge disagreements meaningful?
- Which judge prompt is most reliable?

---

## 19. Step 14: Reliability Tests

### 19.1 Self-consistency

Run the same judge prompt multiple times.

```text
temperature = 0.2
n = 3 or 5
```

Track:

- Majority pass/fail
- Median score
- Score variance
- Confidence variance

Flag unstable examples where repeated judging flips pass/fail.

### 19.2 Prompt sensitivity

Compare:

- Strict judge prompt
- Neutral judge prompt
- No-reference prompt
- Reference-aware prompt

Large changes indicate the rubric is underspecified.

### 19.3 Model sensitivity

Compare multiple judge models.

Examples:

- GPT-4.1 / GPT-4.1 mini
- Claude Sonnet
- Gemini Pro
- Local Qwen / Llama judge

### 19.4 Pairwise position bias

For pairwise judging, run both orders:

```text
A = old agent, B = new agent
A = new agent, B = old agent
```

If the judge often prefers the first output regardless of quality, it has position bias.

---

## 20. Minimal First Milestone

Implement the smallest working version first.

Deliverables:

```text
agent_runs_10.jsonl
judge_results_no_reference.jsonl
judge_results_with_reference.jsonl
agreement_report.md
failure_examples.md
```

Steps:

1. Run 10 InfiAgent-DABench tasks.
2. Save final answer, code, stdout, stderr, and built-in result.
3. Extract dataset summaries.
4. Run deterministic pre-checks.
5. Run no-reference judge.
6. Run reference-aware judge.
7. Compare judge results with built-in evaluator.
8. Manually inspect disagreements.
9. Revise rubric and prompts.

---

## 21. Minimal End-to-End Pseudocode

```python
def main():
    runs = load_jsonl("data/agent_runs.jsonl")
    results = []

    for run in runs:
        dataset_summary = summarize_csv(run["csv_path"])
        prechecks = run_prechecks(run, dataset_summary)

        judge_input = {
            "question": run["question"],
            "dataset_summary": dataset_summary,
            "precheck_results": prechecks,
            "agent_final_answer": run["agent_final_answer"],
            "agent_code": run["agent_code"],
            "stdout": run["stdout"],
            "stderr": run["stderr"],
            "reference_answer": run.get("reference_answer"),
            "builtin_result": run.get("builtin_pass")
        }

        prompt = build_prompt(judge_input, mode="with_reference")
        raw_response = call_llm_judge(prompt)
        judge_result = parse_and_validate(raw_response)

        process_score = compute_process_quality_score(judge_result)
        computed_pass = compute_overall_pass(judge_result)

        classification = classify_run(
            builtin_pass=run.get("builtin_pass"),
            judge_pass=computed_pass,
            process_score=process_score
        )

        results.append({
            "task_id": run["task_id"],
            "builtin_pass": run.get("builtin_pass"),
            "judge_result": judge_result,
            "process_score": process_score,
            "computed_pass": computed_pass,
            "classification": classification
        })

    save_jsonl(results, "data/judge_results.jsonl")
    generate_agreement_report(results)
```

---

## 22. Real-world Use Case

In real life, reference answers are rarely available. Therefore, the production use case is not answer-key grading.

The production use case is a quality-control layer for data-science agents.

The judge checks:

- Did the code run?
- Did the agent use the right columns?
- Did the analysis match the user request?
- Are the final claims supported by stdout, tables, or plots?
- Did the agent overclaim causality?
- Did it ignore requested filters?
- Is the answer reproducible?
- Should the answer be accepted, retried, or sent to human review?

Decision policy:

```text
Judge pass + high confidence:
  accept answer

Judge fail + high confidence:
  block answer or ask agent to retry

Judge low confidence:
  escalate to human review
```

---

## 23. Summary

Use InfiAgent-DABench as a calibration lab because it gives you reference answers and built-in evaluator results.

Use the production judge without reference answers.

The judge should be artifact-based:

```text
question + dataset summary + code + logs + final answer + pre-checks
```

The most important implementation pieces are:

```text
1. Run artifact collection
2. Dataset schema extractor
3. Deterministic pre-checks
4. Judge prompt templates
5. LLM model caller
6. JSON parser and validator
7. Deterministic scoring layer
8. Agreement analyzer
9. Human calibration set
10. Reliability tests
```

The main value of the judge is not perfect truth. It is to detect unsupported, non-reproducible, or obviously flawed data-science agent outputs before they reach users.
