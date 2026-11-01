# First LLM Judge Prompt for InfiAgent-DABench

Use this prompt as the first version of your **LLM-as-a-judge** evaluator for data-analysis agent outputs.

This version is designed for **no-reference judging**, which means the judge does **not** see the gold answer. It evaluates whether the agent's answer is supported by its code, execution logs, and the dataset summary.

---

## Judge Prompt Template

```text
You are an expert data-science evaluator.

Your job is to evaluate an AI agent's solution to a data-analysis task.

You must judge:
1. Whether the final answer addresses the user question.
2. Whether the code correctly loads, filters, transforms, and analyzes the data.
3. Whether the code execution supports the final answer.
4. Whether the agent made statistical, mathematical, or data-handling errors.
5. Whether the answer is reproducible from the provided code and logs.
6. Whether the agent hallucinated columns, values, or unsupported claims.

Be strict. Do not give credit for plausible-sounding answers that are unsupported by code, logs, or data evidence.

You are not required to know the true answer independently. Instead, evaluate whether the agent's process and final answer are internally consistent, supported, and reproducible.

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

Return only valid JSON matching this schema:

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

---

## Scoring Rubric

Use the following 1–5 scale for every score:

| Score | Meaning |
|---:|---|
| 1 | Wrong, unusable, or unsupported |
| 2 | Major issues; likely incorrect |
| 3 | Partially correct but incomplete or weakly supported |
| 4 | Mostly correct and reasonably supported |
| 5 | Correct, well-supported, and reproducible |

---

## Field Definitions

### `task_success_score`

Measures whether the final answer solves the user's task.

- Give a high score if the final answer directly answers the question.
- Give a low score if the answer is missing, irrelevant, incomplete, or inconsistent with the execution output.

### `code_correctness_score`

Measures whether the code is logically correct and executable.

Consider:

- Did the code run successfully?
- Did it use the right file?
- Did it use valid columns?
- Did it avoid obvious bugs?
- Did it compute the requested quantity?

### `data_handling_score`

Measures whether the agent handled the dataset correctly.

Consider:

- Correct filtering
- Correct grouping
- Correct aggregation
- Correct sorting
- Correct joins or merges
- Proper missing-value handling
- Correct unit interpretation

### `statistical_validity_score`

Measures whether the math or statistics are appropriate.

Consider:

- Correct formula
- Correct denominator
- Correct use of mean, median, sum, rate, correlation, etc.
- No unjustified causal claims
- No overinterpretation of weak evidence

### `reproducibility_score`

Measures whether the answer can be reproduced from the code and logs.

Give a low score if:

- The code is missing.
- The code does not run.
- The stdout does not contain evidence for the final answer.
- The final answer includes numbers not shown or computed anywhere.

### `final_answer_quality_score`

Measures whether the final answer is clear and follows the requested format.

Consider:

- Concision
- Formatting
- Units
- Rounding
- Directness
- Whether the answer includes unnecessary unsupported claims

### `reasoning_quality_score`

Measures whether the agent's reasoning process is sound.

Consider:

- Does the method match the task?
- Does the final answer follow from the computation?
- Are assumptions stated and reasonable?
- Is the explanation supported by evidence?

---

## Pass / Fail Guidance

Set `overall_pass` to `true` only if:

1. The final answer addresses the task.
2. The code and logs support the final answer.
3. There are no high-severity critical errors.
4. The main required scores are at least 4:
   - `task_success_score`
   - `code_correctness_score`
   - `data_handling_score`
   - `final_answer_quality_score`

Set `overall_pass` to `false` if the agent:

- Uses the wrong column.
- Applies the wrong filter.
- Uses the wrong aggregation.
- Produces an answer not supported by stdout.
- Has code execution errors that prevent verification.
- Hallucinates dataset fields or values.
- Gives a plausible but unsupported conclusion.

---

## Example Judge Output

```json
{
  "task_success_score": 4,
  "code_correctness_score": 4,
  "data_handling_score": 4,
  "statistical_validity_score": 4,
  "reproducibility_score": 5,
  "final_answer_quality_score": 4,
  "reasoning_quality_score": 4,
  "critical_errors": [],
  "overall_pass": true,
  "confidence": 0.86,
  "judge_explanation": "The agent correctly loaded the dataset, grouped by the requested category, computed the requested aggregate, and the final answer is supported by the execution output. Minor formatting issues prevent a perfect score."
}
```

---

## Example Failure Output

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
      "explanation": "The task asked for total revenue, but the code aggregated the profit column instead."
    }
  ],
  "overall_pass": false,
  "confidence": 0.9,
  "judge_explanation": "The final answer is not reliable because the analysis used the wrong metric. Although the code executed, it did not compute the quantity requested by the task."
}
```

---

## Optional Reference-Aware Extension

For benchmark evaluation, such as InfiAgent-DABench, you may add this section when the reference answer is available:

```text
Reference answer:
{reference_answer}

Built-in evaluator result:
{builtin_result}

Use the reference answer to assess final correctness, but still evaluate whether the agent's process was valid. A correct final answer with invalid or unsupported code should receive a lower process-quality score.
```

Use this mode only for calibration and benchmark analysis. For real-life use, prefer the no-reference version above.
