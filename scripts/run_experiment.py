#!/usr/bin/env python
"""
Run GPT-4o on the DABench benchmark.

Usage:
    python scripts/run_experiment.py [--split dev] [--n 50] [--output data/results/run1]

Reads API keys from .env and benchmark config from config.yaml.
Results are appended to <output>/agent_responses.jsonl (resume-safe).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GPT-4o on DABench.")
    parser.add_argument("--split", default=None, help="Dataset split (default: from config.yaml)")
    parser.add_argument("--n", type=int, default=None, help="Number of tasks to run (default: all)")
    parser.add_argument("--output", default=None, help="Output directory (default: from config.yaml)")
    args = parser.parse_args()

    # Load environment variables (.env file)
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        print("Warning: python-dotenv not installed; skipping .env load.", file=sys.stderr)

    # Load config
    import yaml
    with open(REPO_ROOT / "config.yaml") as fh:
        cfg = yaml.safe_load(fh)

    split = args.split or cfg["experiment"]["split"]
    output_dir = Path(args.output) if args.output else REPO_ROOT / cfg["experiment"]["output_dir"]

    agent_cfg = cfg["agent"]
    model = agent_cfg["model"]
    max_attempts = agent_cfg["max_attempts"]
    timeout = agent_cfg["timeout"]

    # Load tasks
    from src.benchmark.loader import DABenchLoader
    loader = DABenchLoader(REPO_ROOT / "data" / "dabench")
    print(f"Loading tasks (split={split!r})...")
    tasks = loader.load_tasks(split)
    if args.n is not None:
        tasks = tasks[: args.n]
    print(f"  {len(tasks)} tasks to process.")

    # Create agent and runner
    from src.agents.openai_agent import OpenAIAgent
    from src.experiments.runner import ExperimentRunner

    agent = OpenAIAgent(model=model, max_attempts=max_attempts, timeout=timeout)
    run_dir = output_dir / f"{model}_{split}"
    run_dir.mkdir(parents=True, exist_ok=True)

    runner = ExperimentRunner(agent, run_dir)

    print(f"Output directory: {run_dir}")
    print(f"Running {len(tasks)} tasks with model={model!r}  max_attempts={max_attempts}  timeout={timeout}s")

    responses = runner.run(tasks, show_progress=True)

    n_with_answer = sum(1 for r in responses if r.answer)
    n_errors = sum(1 for r in responses if r.error)
    print(f"\nDone. {len(responses)} responses saved.")
    print(f"  Answered:       {n_with_answer}/{len(responses)}")
    print(f"  Errored:        {n_errors}/{len(responses)}")
    print(f"  Results file:   {runner.output_path}")


if __name__ == "__main__":
    main()
