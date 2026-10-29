#!/usr/bin/env python3
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""
Generate Semantic Regex Interpretations for SAE Features

Loads top N features from a state dict and generates semantic regex interpretations
using the Neuronpedia dataset and semantic-regex package.

Usage:
    uv run scripts/interpret_sae_features.py \\
        extras/steering_statedicts/qwen-2.5-7b-it-model.layers.15.statedict \\
        --model-id qwen2.5-7b-it \\
        --layer 15 \\
        --top 20 \\
        --output interpretations.json

Example:
    uv run scripts/interpret_sae_features.py \\
        extras/steering_statedicts/qwen-2.5-7b-it-model.layers.15.statedict \\
        --model-id qwen2.5-7b-it \\
        --layer 15 \\
        --top 50

    python scripts/interpret_sae_features.py \\
        extras/steering_statedicts/qwen-2.5-7b-it-model.layers.15.statedict \\
        --model-id qwen2.5-7b-it \\
        --layer 15 \\
        --top 2

    python scripts/interpret_sae_features.py \
      extras/steering_statedicts/llama-3.1-8b-it-2k-model.layers.17.statedict \
      --model-id llama3.1-8b \
      --layer 17 \
      --hook-point llamascope-res-131k \
      --top 25 \
      --model gpt-5.1 \
      --n-tokens-per-sample 1
"""

import json
import sys
from pathlib import Path

import click
import torch
from tqdm import tqdm

# Import SAE feature utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from notebooks.sae_feature_utils import load_steering_state

# Import semantic-regex
try:
    from semantic_regex import (
        get_neuronpedia_data,
        generate_semantic_regex,
        generate_semantic_regex_prompt,
    )
except ImportError:
    print("ERROR: semantic-regex package not installed")
    print("Install with: uv add semantic-regex")
    sys.exit(1)


def get_top_features(state: dict, top_n: int) -> list[dict]:
    """
    Get top N features preserving the order from the state dict.

    Features are already ranked in the state dict from the selection process,
    so we just take the first top_n without re-sorting.

    Returns list of dicts with feature_id, steering_strength, rank.
    """
    # Convert to float32 to handle BFloat16
    feature_ids = state["mul_indices"].to(torch.long).numpy()
    add_vals = state["add_values"].to(torch.float32).numpy()

    # Take first top_n features, preserving order from state dict
    features = []
    for rank, (fid, strength) in enumerate(zip(feature_ids[:top_n], add_vals[:top_n])):
        features.append(
            {
                "rank": rank,
                "feature_id": int(fid),
                "steering_strength": float(strength),
            }
        )

    return features


def interpret_feature(
    feature_id: int,
    model_id: str,
    layer_spec: str,
    lm=None,
    verbose: bool = False,
    activation_threshold: float = 0.1,
    n_examples: int = 10,
    n_tokens_per_sample: int = 32,
) -> dict:
    """
    Generate semantic regex interpretation for a feature.

    Args:
        feature_id: SAE feature index
        model_id: Neuronpedia model ID (e.g., "qwen2.5-7b-it")
        layer_spec: Full layer specifier (e.g., "15-resid-post-aa")
        lm: Optional DSPy language model instance

    Returns:
        Dict with interpretation results or error info
    """
    try:
        # Step 1: Get feature data from Neuronpedia
        # Returns (batch_tokens, batch_activations) tuple
        if verbose:
            click.echo(f"    Fetching Neuronpedia data for feature {feature_id}...")

        batch_tokens, batch_activations = get_neuronpedia_data(
            model_id, layer_spec, feature_id
        )

        if verbose:
            click.echo(f"    Got {len(batch_tokens)} token batches")
            click.echo(f"    Batch tokens type: {type(batch_tokens)}")
            click.echo(f"    Batch activations type: {type(batch_activations)}")

            # Show structure of first batch
            if len(batch_tokens) > 0:
                click.echo(f"    First batch tokens length: {len(batch_tokens[0])}")
                click.echo(
                    f"    First batch activations length: {len(batch_activations[0])}"
                )
                click.echo(f"    Sample tokens: {batch_tokens[0][:10]}")
                click.echo(f"    Sample activations: {batch_activations[0][:10]}")

                # Check activation value ranges
                all_activations = [act for batch in batch_activations for act in batch]
                if all_activations:
                    click.echo(
                        f"    Activation range: [{min(all_activations):.4f}, {max(all_activations):.4f}]"
                    )
                    click.echo(
                        f"    Activations > {activation_threshold}: {sum(1 for a in all_activations if a > activation_threshold)}/{len(all_activations)}"
                    )
            else:
                click.echo("    WARNING: No batches returned from Neuronpedia!")

        # Step 2: Generate prompt from tokens and activations
        if verbose:
            click.echo(f"    Generating semantic regex prompt...")
            click.echo(
                f"    NOTE: activation_threshold is RELATIVE (0.0-1.0), not absolute"
            )
            click.echo(
                f"    It will be multiplied by max_activation to get actual threshold"
            )

        prompt_data = generate_semantic_regex_prompt(
            batch_tokens=batch_tokens,
            batch_activations=batch_activations,
            activation_threshold=activation_threshold,
            n_data_examples=n_examples,
            n_tokens_per_sample=n_tokens_per_sample,
        )

        if verbose:
            click.echo(
                f"    Prompt generated with {len(prompt_data.get('examples', []))} examples"
            )

            # Show the full prompt data structure
            if "examples" in prompt_data:
                click.echo(f"    Prompt data keys: {list(prompt_data.keys())}")
                if isinstance(prompt_data.get("examples"), str):
                    examples_text = prompt_data["examples"]
                    if examples_text.strip():
                        click.echo(f"    Examples text preview (first 500 chars):")
                        click.echo(f"    {examples_text[:500]}")

            if len(prompt_data.get("examples", [])) == 0 or (
                isinstance(prompt_data.get("examples"), str)
                and not prompt_data["examples"].strip()
            ):
                click.echo(
                    f"    WARNING: No examples generated! This means no {n_tokens_per_sample}-token windows passed filtering."
                )
                click.echo(f"    Possible causes:")
                click.echo(f"      - All snippets were duplicates (common with HTML)")
                click.echo(
                    f"      - activation_threshold too high (current: {activation_threshold})"
                )
                click.echo(
                    f"      - n_tokens_per_sample too large (current: {n_tokens_per_sample})"
                )
                click.echo(
                    f"    Try: --activation-threshold 0.0 --n-tokens-per-sample 8"
                )

        # Step 3: Generate semantic regex using LM
        if verbose:
            click.echo("    Calling LM to generate interpretation...")
            # Show a sample of the prompt being sent
            if "prompt" in prompt_data:
                prompt_text = str(prompt_data["prompt"])
                if len(prompt_text) > 500:
                    click.echo("    Prompt preview (first 500 chars):")
                    click.echo(f"    {prompt_text[:500]}")
                    click.echo("    ...")
                else:
                    click.echo(f"    Full prompt: {prompt_text}")

        try:
            result = generate_semantic_regex(
                prompt_data=prompt_data,
                lm=lm,
                # temperature=1.0,
            )

            if verbose:
                click.echo(f"    Raw result type: {type(result)}")
                click.echo(f"    Raw result: {result}")

            # Extract serializable fields from result
            interpretation = result
            if isinstance(result, dict):
                interpretation = (
                    result.get("description") or result.get("pattern") or str(result)
                )
            else:
                interpretation = str(result)

        except Exception as e:
            # The semantic-regex parser expects 'SR: ' prefix
            # If the error is about missing prefix, try to extract the bare pattern
            error_str = str(e)
            if "Could not find 'SR: ' in response:" in error_str:
                # Extract the raw pattern from the error message
                bare_pattern = error_str.split("Could not find 'SR: ' in response: ")[1]
                if verbose:
                    click.echo(
                        f"    LM returned bare pattern without 'SR: ' prefix: {bare_pattern}"
                    )
                    click.echo("    Accepting bare pattern as valid interpretation")
                interpretation = bare_pattern
            else:
                # Some other error, re-raise it
                raise

        if verbose:
            click.echo(f"    Final interpretation: {interpretation}")

        return {
            "success": True,
            "feature_id": feature_id,
            "interpretation": interpretation,
        }
    except Exception as e:
        import traceback

        error_info = {
            "success": False,
            "feature_id": feature_id,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

        # Try to extract useful debugging info from the error message
        error_str = str(e)
        if "Could not find 'SR: ' in response:" in error_str:
            # Extract the raw response that failed to parse
            try:
                raw_response = error_str.split("Could not find 'SR: ' in response: ")[1]
                error_info["raw_lm_response"] = raw_response
                error_info["note"] = "LM response missing expected 'SR: ' prefix"
            except:
                pass

        if verbose:
            click.echo(f"    ERROR: {error_str}")

        return error_info


@click.command()
@click.argument(
    "state_dict_path",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--model-id",
    required=True,
    help="Neuronpedia model ID (e.g., 'qwen2.5-7b-it')",
)
@click.option(
    "--layer",
    type=int,
    required=True,
    help="Layer number",
)
@click.option(
    "--hook-point",
    default="resid-post-aa",
    help="Hook point ID (default: resid-post-aa)",
)
@click.option(
    "--top",
    type=int,
    default=20,
    help="Number of top features to interpret (default: 20)",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output JSON file (default: {state_dict_name}_interpretations.json)",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Print detailed progress",
)
@click.option(
    "--openai-api-key",
    envvar="OPENAI_API_KEY",
    help="OpenAI API key (or set OPENAI_API_KEY env var)",
)
@click.option(
    "--model",
    default="gpt-4o-mini",
    help="OpenAI model to use for interpretation (default: gpt-4o-mini)",
)
@click.option(
    "--activation-threshold",
    type=float,
    default=0.0,
    help="Minimum activation threshold for highlighting tokens (default: 0.1)",
)
@click.option(
    "--n-examples",
    type=int,
    default=10,
    help="Number of examples to include in prompt (default: 10)",
)
@click.option(
    "--n-tokens-per-sample",
    type=int,
    default=16,
    help="Number of consecutive tokens per example (default: 16, lower = easier to find examples)",
)
def main(
    state_dict_path: Path,
    model_id: str,
    layer: int,
    hook_point: str,
    top: int,
    output: Path | None,
    verbose: bool,
    openai_api_key: str | None,
    model: str,
    activation_threshold: float,
    n_examples: int,
    n_tokens_per_sample: int,
):
    """Generate semantic regex interpretations for SAE features.

    Loads top N features from STATE_DICT_PATH and generates interpretations
    using the Neuronpedia dataset.
    """
    # Load state dict
    click.echo(f"Loading state dict from: {state_dict_path}")
    state = load_steering_state(state_dict_path)
    click.echo(f"✓ Loaded {len(state['mul_indices'])} features")

    # Get top N features
    click.echo(f"\nExtracting top {top} features by steering strength...")
    top_features = get_top_features(state, top)
    click.echo(f"✓ Extracted {len(top_features)} features")

    # Generate interpretations
    click.echo("\nGenerating semantic regex interpretations...")
    click.echo(f"  Model: {model_id}")
    click.echo(f"  Layer spec: {layer}-{hook_point}")
    click.echo(f"  Activation threshold: {activation_threshold}")
    click.echo(f"  Max examples per feature: {n_examples}")
    click.echo(f"  Tokens per sample: {n_tokens_per_sample}")

    # Initialize DSPy language model if API key provided
    lm = None
    if openai_api_key:
        try:
            import dspy

            lm = dspy.LM(model=f"openai/{model}", api_key=openai_api_key)
            click.echo(f"  LM: {model} (via OpenAI)")
        except ImportError:
            click.echo("  WARNING: dspy not installed, running without LM")
        except Exception as e:
            click.echo(f"  WARNING: Failed to initialize LM: {e}")
    else:
        click.echo("  WARNING: No OpenAI API key provided")
        click.echo("  Set OPENAI_API_KEY env var or use --openai-api-key option")
        click.echo("  Interpretations may fail without LM configuration")

    # Combine layer and hook_point into full layer specifier
    layer_spec = f"{layer}-{hook_point}"

    results = []
    for feature in tqdm(top_features, desc="Interpreting features"):
        result = interpret_feature(
            feature["feature_id"],
            model_id,
            layer_spec,
            lm=lm,
            verbose=verbose,
            activation_threshold=activation_threshold,
            n_examples=n_examples,
            n_tokens_per_sample=n_tokens_per_sample,
        )

        # Add feature metadata
        result["rank"] = feature["rank"]
        result["steering_strength"] = feature["steering_strength"]

        results.append(result)

        if verbose and result["success"]:
            click.echo(f"\n  Rank {result['rank']}: Feature {result['feature_id']}")
            click.echo(f"    Interpretation: {result['interpretation']}")
        elif verbose and not result["success"]:
            click.echo(
                f"\n  Rank {result['rank']}: Feature {result['feature_id']} - FAILED"
            )
            click.echo(f"    Error: {result['error']}")

    # Count successes/failures
    successes = sum(1 for r in results if r["success"])
    failures = len(results) - successes

    click.echo(f"\n{'='*60}")
    click.echo("INTERPRETATION RESULTS")
    click.echo(f"{'='*60}")
    click.echo(f"  Total features: {len(results)}")
    click.echo(f"  Successful: {successes}")
    click.echo(f"  Failed: {failures}")

    # Determine output path
    if output is None:
        base_name = state_dict_path.stem
        # Include parameters in filename: model, top_n, n_tokens_per_sample
        model_name_clean = model.replace("/", "-").replace(".", "_")
        output_path = (
            state_dict_path.parent
            / f"{base_name}_interpretations_top{top}_model-{model_name_clean}_tokens{n_tokens_per_sample}.json"
        )
    else:
        output_path = output

    # Save results
    output_data = {
        "metadata": {
            "state_dict_path": str(state_dict_path),
            "model_id": model_id,
            "layer": layer,
            "hook_point": hook_point,
            "layer_spec": layer_spec,
            "top_n": top,
            "total_features": len(results),
            "successful": successes,
            "failed": failures,
        },
        "interpretations": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    click.echo(f"\n✓ Saved results to: {output_path}")
    click.echo(f"{'='*60}")

    # Show sample interpretations
    if successes > 0:
        click.echo("\nSample interpretations (top 3):")
        for result in results[:3]:
            if result["success"]:
                click.echo(f"\n  Rank {result['rank']}: Feature {result['feature_id']}")
                click.echo(f"    Strength: {result['steering_strength']:.6f}")
                click.echo(f"    Pattern: {result['interpretation']}")

    # Show issues if any failures
    if failures > 0:
        click.echo("\n⚠️  NOTES ON FAILURES:")
        parsing_failures = sum(
            1
            for r in results
            if not r["success"]
            and "Could not find 'SR: ' in response" in r.get("error", "")
        )
        if parsing_failures > 0:
            click.echo(
                f"  • {parsing_failures} features failed due to LM response format"
            )
            click.echo(
                "  • The LM is generating semantic regex patterns but without 'SR: ' prefix"
            )
            click.echo("  • Consider trying a different model or adjusting temperature")
            click.echo(
                "  • Raw LM responses are saved in the output JSON for inspection"
            )


if __name__ == "__main__":
    main()
