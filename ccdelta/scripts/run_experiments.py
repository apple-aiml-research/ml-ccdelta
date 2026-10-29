# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""Run all CC-Delta experiments: baseline, CC-Delta (SAE), and CAA (dense).

By default runs all three stages. Use --stages to select specific ones.
All outputs go under a single --results-dir (default: ./results).

Examples:
    # Run everything
    python -m ccdelta.scripts.run_experiments

    # Run only baseline and CC-Delta
    python -m ccdelta.scripts.run_experiments --stages baseline ccdelta

    # Run only CAA
    python -m ccdelta.scripts.run_experiments --stages caa

    # Override the model
    python -m ccdelta.scripts.run_experiments --model gemma-2-9b-it

    # Run on the full dataset (disable fast mode)
    python -m ccdelta.scripts.run_experiments --full

    # Run OR-Bench over-refusal evaluation
    python -m ccdelta.scripts.run_experiments --eval-suite orbench

    # Run PAIR adaptive attack evaluation
    python -m ccdelta.scripts.run_experiments --eval-suite pair

    # Custom output directory (rm -rf to clean up)
    python -m ccdelta.scripts.run_experiments --results-dir ./my_results
"""

import argparse
import os
import subprocess
import sys

ALL_STAGES = ["baseline", "ccdelta", "caa"]

DEFAULT_RESULTS_DIR = os.path.join(os.environ.get("OUTPUT_DIR", "."), "results")

EVAL_SUITES = {
    "jailbreak": [],
    "orbench": [
        "evaluation=[text-generation,orbench,strongreject_finetuned,strongreject_rubric]",
        "task_params=or_bench",
    ],
    "pair": [
        "evaluation=[pair,strongreject_finetuned,strongreject_rubric,fluency]",
    ],
}

# CC-Delta sweep parameters
CCDELTA_NUM_FEATURES = [500]
# CCDELTA_STEERING_MULTIPLIERS = [0.4, 0.8, 1.0]
CCDELTA_STEERING_MULTIPLIERS = [0.4]

# CAA sweep parameters
# CAA_STRENGTHS = [0.4, 0.8, 1.0]
CAA_STRENGTHS = [0.4]

STAGE_CONFIGS = {
    "baseline": {
        "config_name": "text_generation_baseline",
        "default_model": "gemma-2-2b-it",
        "results_subdir": "baseline",
        "description": "Baseline evaluation (no intervention)",
    },
    "ccdelta": {
        "config_name": "text_generation_ccdelta",
        "default_model": "gemma-2-2b-it-sae",
        "results_subdir": "sae",
        "description": "CC-Delta (SAE-based steering)",
    },
    "caa": {
        "config_name": "text_generation_caa",
        "default_model": "gemma-2-2b-it",
        "results_subdir": "dense",
        "description": "CAA (dense contrastive steering)",
    },
}


def _run_pipeline(cmd: list[str], label: str) -> None:
    """Run a single pipeline command, exit on failure."""
    result = subprocess.run(cmd)  # nosemgrep
    if result.returncode != 0:
        print(f"\nERROR: {label} failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def run_baseline(
    results_dir: str,
    model: str | None = None,
    full: bool = False,
    extra_overrides: list[str] | None = None,
    eval_suite_overrides: list[str] | None = None,
) -> None:
    """Run baseline (no intervention)."""
    cfg = STAGE_CONFIGS["baseline"]
    model = model or cfg["default_model"]
    outdir = f"{results_dir}/{cfg['results_subdir']}"

    print(f"\n{'=' * 60}")
    print(f"Running: {cfg['description']}")
    print(f"  model:  {model}")
    print(f"  output: {outdir}")
    print(f"{'=' * 60}\n")

    cmd = [
        sys.executable,
        "-m",
        "ccdelta.scripts.pipeline",
        "--config-name",
        cfg["config_name"],
        f"model={model}",
        f"results_dir={outdir}",
    ]
    if full:
        cmd.append("fast=false")
    cmd.extend(eval_suite_overrides or [])
    cmd.extend(extra_overrides or [])

    _run_pipeline(cmd, "baseline")


def run_ccdelta(
    results_dir: str,
    model: str | None = None,
    full: bool = False,
    extra_overrides: list[str] | None = None,
    eval_suite_overrides: list[str] | None = None,
) -> None:
    """Run CC-Delta (SAE) sweep over num_features and steering_multiplier."""
    cfg = STAGE_CONFIGS["ccdelta"]
    model = model or cfg["default_model"]

    print(f"\n{'=' * 60}")
    print(f"Running: {cfg['description']}")
    print(f"  model:  {model}")
    print(f"  num_features:          {CCDELTA_NUM_FEATURES}")
    print(f"  steering_multipliers:  {CCDELTA_STEERING_MULTIPLIERS}")
    print(f"{'=' * 60}\n")

    for nf in CCDELTA_NUM_FEATURES:
        for sm in CCDELTA_STEERING_MULTIPLIERS:
            outdir = f"{results_dir}/{cfg['results_subdir']}/nf{nf}_sm{sm}"
            experiment_id = f"num_features:{nf}__steering_multiplier:{sm}"
            print(f"  >> num_features={nf}, steering_multiplier={sm}")

            cmd = [
                sys.executable,
                "-m",
                "ccdelta.scripts.pipeline",
                "--config-name",
                cfg["config_name"],
                f"model={model}",
                f"results_dir={outdir}",
                f"intervention_params.hook_params.num_features={nf}",
                f"intervention_params.hook_params.steering_multiplier={sm}",
                f"text_generation.experiment_id={experiment_id}",
            ]
            if full:
                cmd.append("fast=false")
            cmd.extend(eval_suite_overrides or [])
            cmd.extend(extra_overrides or [])

            _run_pipeline(cmd, f"ccdelta nf={nf} sm={sm}")


def run_caa(
    results_dir: str,
    model: str | None = None,
    full: bool = False,
    extra_overrides: list[str] | None = None,
    eval_suite_overrides: list[str] | None = None,
) -> None:
    """Run CAA (dense) sweep over strength."""
    cfg = STAGE_CONFIGS["caa"]
    model = model or cfg["default_model"]

    print(f"\n{'=' * 60}")
    print(f"Running: {cfg['description']}")
    print(f"  model:     {model}")
    print(f"  strengths: {CAA_STRENGTHS}")
    print(f"{'=' * 60}\n")

    for s in CAA_STRENGTHS:
        outdir = f"{results_dir}/{cfg['results_subdir']}/str{s}"
        experiment_id = f"strength:{s}"
        print(f"  >> strength={s}")

        cmd = [
            sys.executable,
            "-m",
            "ccdelta.scripts.pipeline",
            "--config-name",
            cfg["config_name"],
            f"model={model}",
            f"results_dir={outdir}",
            f"intervention_params.hook_params.strength={s}",
            f"text_generation.experiment_id={experiment_id}",
        ]
        if full:
            cmd.append("fast=false")
        cmd.extend(eval_suite_overrides or [])
        cmd.extend(extra_overrides or [])

        _run_pipeline(cmd, f"caa strength={s}")


STAGE_RUNNERS = {
    "baseline": run_baseline,
    "ccdelta": run_ccdelta,
    "caa": run_caa,
}


def main():
    parser = argparse.ArgumentParser(
        description="Run CC-Delta experiments (baseline, CC-Delta, CAA)."
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=ALL_STAGES,
        default=ALL_STAGES,
        help="Which stages to run (default: all).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the model for all stages (default: per-stage default).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run on the full dataset (disable fast mode).",
    )
    parser.add_argument(
        "--eval-suite",
        type=str,
        choices=list(EVAL_SUITES.keys()),
        default="jailbreak",
        help="Evaluation suite: jailbreak (default), orbench (over-refusal), or pair (adaptive attack).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help=f"Root directory for all outputs (default: {DEFAULT_RESULTS_DIR}).",
    )
    args, extra_overrides = parser.parse_known_args()

    print(f"Stages:      {', '.join(args.stages)}")
    print(f"Eval suite:  {args.eval_suite}")
    print(f"Mode:        {'full dataset' if args.full else 'fast (subset)'}")
    print(f"Results dir: {args.results_dir}")
    if extra_overrides:
        print(f"Overrides:   {' '.join(extra_overrides)}")

    eval_suite_overrides = EVAL_SUITES[args.eval_suite]

    for stage in args.stages:
        STAGE_RUNNERS[stage](
            results_dir=args.results_dir,
            model=args.model,
            full=args.full,
            extra_overrides=extra_overrides,
            eval_suite_overrides=eval_suite_overrides,
        )

    print(f"\n{'=' * 60}")
    print(f"All stages complete. Results in {args.results_dir}/")
    print("Open notebooks/results_analysis.ipynb to analyze results.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
