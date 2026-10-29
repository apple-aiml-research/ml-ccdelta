# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""Run SAE reconstruction error and sparsity analysis.

Measures SAE reconstruction quality (FVU, L0) on paired harmful/jailbreak data.
Two stages:
  - reconstruction: default SAE configs for each model
  - sparsity: sweep over SAE L0 variants (different sparsity levels)

Examples:
    # Run both stages
    python -m ccdelta.scripts.run_sae_analysis

    # Reconstruction error for default SAEs only
    python -m ccdelta.scripts.run_sae_analysis --stages reconstruction

    # Sparsity sweep only
    python -m ccdelta.scripts.run_sae_analysis --stages sparsity

    # Specific sparsity group (also controls reconstruction models)
    python -m ccdelta.scripts.run_sae_analysis --groups gemma-2b-pt

    # Override model for reconstruction stage
    python -m ccdelta.scripts.run_sae_analysis --stages reconstruction --model gemma-2-9b-it-sae

    # Limit data for quick testing
    python -m ccdelta.scripts.run_sae_analysis --max-batches 2

    # Custom output directory
    python -m ccdelta.scripts.run_sae_analysis --results-dir ./my_sae_analysis
"""

import argparse
import subprocess
import sys

ALL_STAGES = ["reconstruction", "sparsity"]

# Default models for the reconstruction stage
RECONSTRUCTION_MODELS = [
    "gemma-2-2b-it-sae",
    "gemma-2-9b-it-sae",
]

# SAE sparsity variant configurations
# Format: {group_name: {model_config, module_name, description, variants: [(release, sae_id, desc)]}}
SPARSITY_GROUPS = {
    "gemma-2b-pt": {
        "model_config": "gemma-2-2b-it-sae",
        "module_name": "model.layers.14",
        "description": "Gemma-2-2B with PT SAEs (layer 14, width 65k)",
        "variants": [
            ("gemma-scope-2b-pt-res", "layer_14/width_65k/average_l0_21", "L0_21"),
            ("gemma-scope-2b-pt-res", "layer_14/width_65k/average_l0_40", "L0_40"),
            ("gemma-scope-2b-pt-res", "layer_14/width_65k/average_l0_73", "L0_73"),
            ("gemma-scope-2b-pt-res", "layer_14/width_65k/average_l0_144", "L0_144"),
            ("gemma-scope-2b-pt-res", "layer_14/width_65k/average_l0_284", "L0_284"),
        ],
    },
    "gemma-9b-it": {
        "model_config": "gemma-2-9b-it-sae",
        "module_name": "model.layers.20",
        "description": "Gemma-2-9B with IT SAEs (layer 20, width 131k)",
        "variants": [
            ("gemma-scope-9b-it-res", "layer_20/width_131k/average_l0_13", "L0_13"),
            ("gemma-scope-9b-it-res", "layer_20/width_131k/average_l0_24", "L0_24"),
            ("gemma-scope-9b-it-res", "layer_20/width_131k/average_l0_43", "L0_43"),
            ("gemma-scope-9b-it-res", "layer_20/width_131k/average_l0_81", "L0_81"),
            ("gemma-scope-9b-it-res", "layer_20/width_131k/average_l0_153", "L0_153"),
        ],
    },
    "gemma-9b-pt": {
        "model_config": "gemma-2-9b-it-sae",
        "module_name": "model.layers.20",
        "description": "Gemma-2-9B with PT SAEs (layer 20, width 131k)",
        "variants": [
            ("gemma-scope-9b-pt-res", "layer_20/width_131k/average_l0_11", "L0_11"),
            ("gemma-scope-9b-pt-res", "layer_20/width_131k/average_l0_34", "L0_34"),
            ("gemma-scope-9b-pt-res", "layer_20/width_131k/average_l0_114", "L0_114"),
            ("gemma-scope-9b-pt-res", "layer_20/width_131k/average_l0_221", "L0_221"),
            ("gemma-scope-9b-pt-res", "layer_20/width_131k/average_l0_276", "L0_276"),
        ],
    },
}

CONFIG_NAME = "text_generation_sae_error_analysis"


def _run_pipeline(cmd: list[str], label: str) -> None:
    """Run a single pipeline command, exit on failure."""
    print(f"\n{'=' * 60}")
    print(f"Running: {label}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(cmd)  # nosemgrep
    if result.returncode != 0:
        print(f"\nERROR: {label} failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def _models_for_groups(groups: list[str] | None) -> list[str]:
    """Derive unique model configs from selected sparsity groups."""
    groups_to_use = groups or list(SPARSITY_GROUPS.keys())
    seen = set()
    models = []
    for g in groups_to_use:
        m = SPARSITY_GROUPS[g]["model_config"]
        if m not in seen:
            seen.add(m)
            models.append(m)
    return models


def run_reconstruction(
    results_dir: str,
    model: str | None = None,
    groups: list[str] | None = None,
    max_batches: int | None = None,
) -> None:
    """Run reconstruction error analysis with default SAE configs."""
    if model:
        models = [model]
    elif groups:
        models = _models_for_groups(groups)
    else:
        models = RECONSTRUCTION_MODELS

    for m in models:
        outdir = f"{results_dir}/reconstruction/{m}"
        cmd = [
            sys.executable,
            "-m",
            "ccdelta.scripts.pipeline",
            "--config-name",
            CONFIG_NAME,
            f"model={m}",
            f"results_dir={outdir}",
            f"cache_dir={outdir}",
        ]
        if max_batches is not None:
            cmd.append(f"responses.max_batches={max_batches}")

        _run_pipeline(cmd, f"reconstruction error: {m}")


def run_sparsity(
    results_dir: str,
    groups: list[str] | None = None,
    max_batches: int | None = None,
) -> None:
    """Run error analysis sweeping over SAE sparsity variants."""
    groups_to_run = groups or list(SPARSITY_GROUPS.keys())

    for group_name in groups_to_run:
        group = SPARSITY_GROUPS[group_name]
        model_config = group["model_config"]
        module_name = group["module_name"]

        print(f"\n{'=' * 60}")
        print(f"Sparsity sweep: {group['description']}")
        print(f"  Variants: {len(group['variants'])}")
        print(f"{'=' * 60}")

        for release, sae_id, variant_desc in group["variants"]:
            outdir = f"{results_dir}/sparsity/{group_name}/{variant_desc}"

            cmd = [
                sys.executable,
                "-m",
                "ccdelta.scripts.pipeline",
                "--config-name",
                CONFIG_NAME,
                f"model={model_config}",
                f"results_dir={outdir}",
                f"cache_dir={outdir}",
                f"+model.sae_configs.{module_name}.release={release}",
                f"+model.sae_configs.{module_name}.sae_id={sae_id}",
            ]
            if max_batches is not None:
                cmd.append(f"responses.max_batches={max_batches}")

            _run_pipeline(cmd, f"sparsity: {group_name}/{variant_desc}")


def main():
    parser = argparse.ArgumentParser(
        description="Run SAE reconstruction error and sparsity analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sparsity groups:
  gemma-2b-pt   Gemma-2-2B with PT SAEs (layer 14, L0: 21, 40, 73, 144, 284)
  gemma-9b-it   Gemma-2-9B with IT SAEs (layer 20, L0: 13, 24, 43, 81, 153)
  gemma-9b-pt   Gemma-2-9B with PT SAEs (layer 20, L0: 11, 34, 114, 221, 276)
        """,
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
        help="Override model for reconstruction stage.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=list(SPARSITY_GROUPS.keys()),
        default=None,
        help="Groups to run. Controls both reconstruction models and sparsity variants (default: all).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results/sae_analysis",
        help="Root directory for outputs (default: ./results/sae_analysis).",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Limit number of data batches (useful for testing).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without executing.",
    )
    args = parser.parse_args()

    print(f"Stages:      {', '.join(args.stages)}")
    print(f"Results dir: {args.results_dir}")
    if args.max_batches:
        print(f"Max batches: {args.max_batches}")

    if args.dry_run:
        if "reconstruction" in args.stages:
            if args.model:
                models = [args.model]
            elif args.groups:
                models = _models_for_groups(args.groups)
            else:
                models = RECONSTRUCTION_MODELS
            print(f"\n[reconstruction] Would run error analysis for: {models}")
        if "sparsity" in args.stages:
            groups = args.groups or list(SPARSITY_GROUPS.keys())
            for g in groups:
                variants = SPARSITY_GROUPS[g]["variants"]
                print(f"\n[sparsity/{g}] Would run {len(variants)} variants:")
                for _, sae_id, desc in variants:
                    print(f"  - {desc}: {sae_id}")
        print("\n[DRY RUN] No jobs executed.")
        return

    if "reconstruction" in args.stages:
        run_reconstruction(
            results_dir=args.results_dir,
            model=args.model,
            groups=args.groups,
            max_batches=args.max_batches,
        )

    if "sparsity" in args.stages:
        run_sparsity(
            results_dir=args.results_dir,
            groups=args.groups,
            max_batches=args.max_batches,
        )

    print(f"\n{'=' * 60}")
    print(f"All stages complete. Results in {args.results_dir}/")
    print("Open notebooks/sae_quality_analysis.ipynb to analyze results.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
